import re
d_format_long = "%Y-%m-%d %H:%M:%S"
d_format_short = "%Y-%m-%d"
d_format_ms = "%Y-%m-%d %H:%M:%S.%f"

#re expression
pattern_number_re = re.compile(r'[0-9]')
pattern_letters_re = re.compile(r'[A-Za-z]')