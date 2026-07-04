# export_contentvec.py — ContentVec feature extractor -> TWO ONNX graphs
#
#   contentvec_768l12.onnx : vec768l12 (RVC v2 + SoVITS 4.1)
#       features = encoder layer 12 output, 768-d, NO final_proj
#   contentvec_256l9.onnx  : vec256l9  (RVC v1 + SoVITS 4.0)
#       features = final_proj(encoder layer 9 output), 256-d
#
# Input : "waveform"  f32 [1, N]   mono 16 kHz raw samples (dynamic N)
# Output: "features"  f32 [1, T, dim]   T = (N - 400) // 320 + 1   (50 fps)
#
# Source checkpoint: fairseq HuBERT-arch ContentVec "legacy_500"
#   D:\MyDev\so-vits-svc\so-vits-svc\pretrain\checkpoint_best_legacy_500.pt
#   (bit-identical file: D:\MyDev\RVC\RVC20240604Nvidia\assets\hubert\hubert_base.pt,
#    md5 b76f784c1958d4e535cd0f6151ca35e4 both)
#
# REFERENCE SEMANTICS (both original pipelines use the SAME call, so one pair of
# onnx serves both backends):
#   so-vits vencoder\ContentVec768L12.py / ContentVec256L9.py:
#       model.extract_features(source, padding_mask=all-False, output_layer=12|9)
#       (+ final_proj for 256)
#   RVC infer\modules\vc\pipeline.py:212-220:
#       output_layer = 9 if v1 else 12; final_proj only for v1  -> identical.
#
# Model classes below are a VERBATIM port of the inference path of fairseq 0.12.2
#   fairseq/models/hubert/hubert.py   (HubertModel.forward, features_only, mask=False)
#   fairseq/models/wav2vec/wav2vec2.py (ConvFeatureExtractionModel, TransformerEncoder,
#                                       TransformerSentenceEncoderLayer, make_conv_pos,
#                                       post-norm branch: layer_norm_first=False)
#   fairseq/modules/multihead_attention.py (q*scaling -> softmax -> out_proj)
# with these EXACT no-op reductions for the reference call signature (all proven
# no-ops for an all-False padding mask in eval mode, see gate_contentvec.py):
#   * padding_mask all-False: index_put(x, mask, 0) is identity; attention
#     key_padding_mask with no True entries == no mask.
#   * required_seq_len_multiple=2 padding: the padded frame is attention-masked
#     for all real queries and sliced off afterwards -> real frames unchanged.
#   * dropout / layerdrop / apply_mask: eval() + mask=False -> inactive.
# The layout is B,T,C instead of fairseq's T,B,C (pure transpose bookkeeping).
#
# Checkpoint weights are fp16 on disk; fairseq load_state_dict casts them into the
# fp32 model exactly like our strict=True load does. pos_conv weight-norm
# (weight_g/weight_v, dim=2) is loaded strict then fused via remove_weight_norm —
# the same torch._weight_norm computation fairseq runs per-forward.
#
# Checkpoint cfg (embedded, verified): encoder_layers=12 embed=768 ffn=3072 heads=12
#   activation=gelu(exact/erf) layer_norm_first=False extractor_mode=default
#   conv_feature_layers=[(512,10,5)]+[(512,3,2)]*4+[(512,2,2)]*2 conv_bias=False
#   conv_pos=128 conv_pos_groups=16 final_dim=256; task: sample_rate=16000
#   normalize=False (no waveform layer_norm — raw samples in).
#
# Gates: converter\verify\voice\gate_contentvec.py (run it after this script).
#
# Usage:
#   .venv\Scripts\python.exe export_contentvec.py [--pt PATH] [--outdir DIR]
#
# torch.onnx.export: dynamo=False, opset 17 (project-wide rule).

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PT = r"D:\MyDev\so-vits-svc\so-vits-svc\pretrain\checkpoint_best_legacy_500.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "test_output")

SAMPLE_RATE = 16000
FPS = 50
FRAME_FORMULA = "(N-400)//320+1"

# architecture constants (from the checkpoint's embedded cfg — see header)
CONV_LAYERS = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
EMBED_DIM = 768
FFN_DIM = 3072
NUM_HEADS = 12
NUM_LAYERS = 12
FINAL_DIM = 256
CONV_POS = 128
CONV_POS_GROUPS = 16


# ------------------------------------------------------------------
# fairseq checkpoint loader (no fairseq install needed):
# the pickle references fairseq.data.dictionary.Dictionary inside task_state;
# stub unknown classes — we only consume ckpt["model"] tensors + ckpt["cfg"] dict.
# ------------------------------------------------------------------
class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            class Stub:
                def __init__(self, *a, **k):
                    pass

                def __setstate__(self, state):
                    self.__dict__["_state"] = state

            Stub.__name__ = name
            return Stub


class _StubPickleModule:
    Unpickler = _StubUnpickler

    @staticmethod
    def load(f, **kw):
        return _StubUnpickler(f, **kw).load()


def load_fairseq_checkpoint(path):
    ck = torch.load(path, map_location="cpu", pickle_module=_StubPickleModule,
                    weights_only=False)
    return ck["model"], ck["cfg"]


# ------------------------------------------------------------------
# VERBATIM inference-path port (fairseq 0.12.2), module/parameter names kept
# IDENTICAL to the checkpoint so load_state_dict(strict=True) round-trips.
# ------------------------------------------------------------------
class SamePad(nn.Module):
    # fairseq/modules/same_pad.py — kernel 128 is even -> trim the last frame
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvFeatureExtractionModel(nn.Module):
    # wav2vec2.py ConvFeatureExtractionModel, mode="default", conv_bias=False:
    # layer 0 = Sequential(Conv1d, Dropout, Fp32GroupNorm(dim,dim), GELU)
    # layers 1..6 = Sequential(Conv1d, Dropout, GELU)
    # (Fp32GroupNorm == nn.GroupNorm for an fp32 graph; Dropout(0) kept only so the
    #  Sequential indices — hence state-dict keys .0/.2 — match the original.)
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_d, dim, k, stride=stride, bias=False)
            if i == 0:
                block = nn.Sequential(conv, nn.Dropout(0.0),
                                      nn.GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                block = nn.Sequential(conv, nn.Dropout(0.0), nn.GELU())
            self.conv_layers.append(block)
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class MultiheadAttention(nn.Module):
    # fairseq/modules/multihead_attention.py, self-attention eval path, no mask:
    # q = q_proj(x) * head_dim**-0.5 ; softmax(q k^T) v ; out_proj
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):  # x: [1, T, C]
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [1, T, C] -> [1, H, T, D]
        q = q.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))       # [1, H, T, T]
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn = torch.matmul(attn_probs, v)                        # [1, H, T, D]
        attn = attn.transpose(1, 2).reshape(1, -1, self.embed_dim)
        return self.out_proj(attn)


class TransformerSentenceEncoderLayer(nn.Module):
    # wav2vec2.py TransformerSentenceEncoderLayer, layer_norm_first=False branch
    # (post-norm), activation_fn=gelu (exact/erf), dropouts eval-inactive.
    def __init__(self, embed_dim, ffn_dim, num_heads):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.self_attn(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class TransformerEncoder(nn.Module):
    # wav2vec2.py TransformerEncoder (layer_norm_first=False):
    #   x = x + pos_conv(x); x = layer_norm(x); run layers, stop after tgt_layer.
    # pos_conv = make_conv_pos(768, 128, 16): weight-normed (dim=2) grouped Conv1d
    #   + SamePad(128) + GELU. weight_norm kept at load time for key parity
    #   (weight_g/weight_v), fused afterwards via remove_weight_norm.
    def __init__(self, embed_dim, ffn_dim, num_heads, num_layers,
                 conv_pos, conv_pos_groups):
        super().__init__()
        pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_pos,
                             padding=conv_pos // 2, groups=conv_pos_groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # weight_norm deprecation
            pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), nn.GELU())
        self.layers = nn.ModuleList(
            TransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads)
            for _ in range(num_layers)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, tgt_layer):  # x: [1, T, C]; tgt_layer: 0-based
        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = self.layer_norm(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == tgt_layer:
                break
        return x


class HubertModel(nn.Module):
    # hubert.py HubertModel — inference members only, but ALL checkpoint tensors
    # present (mask_emb / label_embs_concat unused in forward) so strict=True holds.
    def __init__(self):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.zeros(EMBED_DIM))
        self.label_embs_concat = nn.Parameter(torch.zeros(504, FINAL_DIM))
        self.feature_extractor = ConvFeatureExtractionModel(CONV_LAYERS)
        self.layer_norm = nn.LayerNorm(CONV_LAYERS[-1][0])          # LayerNorm(512)
        self.post_extract_proj = nn.Linear(CONV_LAYERS[-1][0], EMBED_DIM)
        self.encoder = TransformerEncoder(EMBED_DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS,
                                          CONV_POS, CONV_POS_GROUPS)
        self.final_proj = nn.Linear(EMBED_DIM, FINAL_DIM)

    def extract(self, source, output_layer):
        # hubert.py forward(features_only=True, mask=False), all-False padding mask.
        # output_layer is 1-based (fairseq convention: encoder layer = output_layer-1)
        x = self.feature_extractor(source)      # [1, 512, T]
        x = x.transpose(1, 2)                   # [1, T, 512]
        x = self.layer_norm(x)
        x = self.post_extract_proj(x)           # [1, T, 768]
        x = self.encoder(x, tgt_layer=output_layer - 1)
        return x

    def fuse_pos_conv_weight_norm(self):
        # torch._weight_norm(v, g, 2) — the exact tensor fairseq computes per-forward
        nn.utils.remove_weight_norm(self.encoder.pos_conv[0])


class Vec768L12Export(nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    def forward(self, waveform):                # [1, N] -> [1, T, 768]
        return self.hubert.extract(waveform, output_layer=12)


class Vec256L9Export(nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    def forward(self, waveform):                # [1, N] -> [1, T, 256]
        return self.hubert.final_proj(self.hubert.extract(waveform, output_layer=9))


# ------------------------------------------------------------------
def build_model(pt_path, fuse_pos_conv=True):
    # fuse_pos_conv=False keeps the live weight-norm reparametrization (weight
    # recomputed from g/v each forward, in the model's dtype) — needed by the f64
    # structural gate; export always fuses (fp32 constant, same tensor fairseq
    # materializes in fp32 inference).
    sd, cfg = load_fairseq_checkpoint(pt_path)
    # sanity: the embedded cfg must match the constants this port hardcodes
    mc = cfg["model"]
    assert mc["encoder_layers"] == NUM_LAYERS and mc["encoder_embed_dim"] == EMBED_DIM
    assert mc["encoder_ffn_embed_dim"] == FFN_DIM
    assert mc["encoder_attention_heads"] == NUM_HEADS
    assert mc["activation_fn"] == "gelu" and mc["layer_norm_first"] is False
    assert mc["extractor_mode"] == "default" and mc["conv_bias"] is False
    assert eval(mc["conv_feature_layers"]) == CONV_LAYERS
    assert mc["conv_pos"] == CONV_POS and mc["conv_pos_groups"] == CONV_POS_GROUPS
    assert mc["final_dim"] == FINAL_DIM
    assert cfg["task"]["normalize"] is False and cfg["task"]["sample_rate"] == SAMPLE_RATE

    model = HubertModel()
    model.load_state_dict(sd, strict=True)      # fp16 ckpt -> fp32 params (cast copy)
    model.eval()
    if fuse_pos_conv:
        model.fuse_pos_conv_weight_norm()
    return model


def expected_frames(n):
    return (n - 400) // 320 + 1


def export_one(model_cls, hubert, out_path, dim, variant):
    wrapper = model_cls(hubert).eval()
    dummy = torch.randn(1, 32000, dtype=torch.float32)
    with torch.no_grad():
        ref_out = wrapper(dummy)
    assert ref_out.shape == (1, expected_frames(32000), dim), ref_out.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            input_names=["waveform"],
            output_names=["features"],
            dynamic_axes={"waveform": {1: "num_samples"},
                          "features": {1: "num_frames"}},
            opset_version=17,
            dynamo=False,
        )

    sidecar = {
        "type": "contentvec",
        "variant": variant,
        "dim": dim,
        "sample_rate": SAMPLE_RATE,
        "fps": FPS,
        "frame_formula": FRAME_FORMULA,
        "input": {"name": "waveform", "dtype": "float32", "shape": [1, "num_samples"]},
        "output": {"name": "features", "dtype": "float32",
                   "shape": [1, "num_frames", dim]},
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    # smoke: ORT CPU, two lengths (odd + even frame counts), shape + parity vs torch
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    for n in (32000, 48117):
        wav = torch.randn(1, n, dtype=torch.float32)
        with torch.no_grad():
            want = wrapper(wav).numpy()
        got = sess.run(None, {"waveform": wav.numpy()})[0]
        assert got.shape == (1, expected_frames(n), dim), got.shape
        mad = float(np.abs(got - want).max())
        print(f"  [{variant}] N={n}: T={got.shape[1]} onnx-vs-torch max_abs={mad:.3e}")
        assert mad < 1e-4, mad
    print(f"  [{variant}] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=DEFAULT_PT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"loading {args.pt}")
    hubert = build_model(args.pt)

    # empirical frame-formula check on the conv stack alone
    with torch.no_grad():
        for n in (400, 401, 719, 720, 16000, 16001, 31999, 160000):
            t = hubert.feature_extractor(torch.zeros(1, n)).shape[2]
            assert t == expected_frames(n), (n, t, expected_frames(n))
    print(f"frame formula {FRAME_FORMULA} verified empirically (8 lengths)")

    export_one(Vec768L12Export, hubert,
               os.path.join(args.outdir, "contentvec_768l12.onnx"), 768, "vec768l12")
    export_one(Vec256L9Export, hubert,
               os.path.join(args.outdir, "contentvec_256l9.onnx"), 256, "vec256l9")
    print("done - now run converter\\verify\\voice\\gate_contentvec.py")


if __name__ == "__main__":
    main()
