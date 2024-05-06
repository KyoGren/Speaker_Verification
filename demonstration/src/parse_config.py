class Dict2dot():
    def __init__(self, input_dict = dict()):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = Dict2dot(value)
            setattr(self, key, value)

config_data = {
  "task": "",
  "data": {
    "TD_SV_data":{
      "sr": 16000,
      "nfft": 512,
      "window": 0.025,
      "hop": 0.01,
      "nmels": 40,
      "frame": 80,
      "duration": 1,
      "enroll_path": "./demonstration/data/tdsv/enrollment_audio.wav",
      "eval_path": "./demonstration/data/tdsv/evaluation_audio.wav"
    },
    "TI_SV_data": {
      "sr": 16000,
      "nfft": 512,
      "window": 0.025,
      "hop": 0.01,
      "nmels": 40,
      "frame_low": 160,
      "frame": 180,
      "duration": 2.05,
      "enroll_path": "./demonstration/data/tisv/enrollment_audio.wav",
      "eval_path": "./demonstration/data/tisv/evaluation_audio.wav"
    }
  },
  "model": {
    "TD_SV_model": {
      "hidden": 128,
      "num_layer": 3,
      "proj": 64,
      "final_model_path": "./checkpoint/tdsv/final_model.model",
      "optim_model_path": "./checkpoint/tdsv/optim_model.model",
      "loss": "contrast"
    },
    "TI_SV_model": {
      "hidden": 768,
      "num_layer": 3,
      "proj": 256,
      "final_model_path": "./checkpoint/tisv/final_model.model",
      "optim_model_path": "./checkpoint/tisv/optim_model.model",
      "loss": "softmax"
    }
  },
}

config= Dict2dot(config_data)