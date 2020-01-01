"""
@author: penhe@microsoft.com
"""

class InitSpec:
  def __init__(self, fields):
    # initialize from random
    self.name = fields[0]
    self.use_pretrain = True if len(fields)<2 else fields[1].lower()!='false'
    self.freeze_ratio = float(fields[2]) if len(fields)>2 else 0
    self.lr_decay = min(float(fields[3]) if len(fields)>3 else 1, 1)
    self.mapping = fields[4] if len(fields)>4 else None

  @classmethod
  def load(cls, init_spec):
    with open(init_spec, 'r', encoding='utf8') as fs:
      lines = [l.strip().split() for l in fs.readlines() if not (l.strip().startswith('#') or len(l.strip())<5)]
      var_specs = {l[0]:cls([f.strip() for f in l]) for l in lines}
      return var_specs
        # TODO: add fuzz match
