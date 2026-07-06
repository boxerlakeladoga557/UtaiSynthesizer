# Vendored verbatim from so-vits-svc 4.1-Stable (modules/F0Predictor/F0Predictor.py @ 730930d).
# Changes vs upstream: package-relative imports only.
class F0Predictor(object):
    def compute_f0(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        '''
        pass

    def compute_f0_uv(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        '''
        pass