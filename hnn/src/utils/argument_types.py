
def boolean_string(s):
  if s.lower() not in {'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s.lower() == 'true'
