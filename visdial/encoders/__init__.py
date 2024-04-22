from visdial.encoders.svg.svg import SVGEncoder

def Encoder(hparams, *args):
  name_enc_map = {
    "svg": SVGEncoder,  # Ours
  }
  return name_enc_map[hparams.encoder](hparams, *args)