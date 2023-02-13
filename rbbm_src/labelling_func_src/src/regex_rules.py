import re 
from lark import Lark, Token
from classes import make_regex_lf
import glob
from lark.exceptions import UnexpectedEOF,UnexpectedCharacters
assasin_dir = "/home/jayli/Desktop/updates_spamassassin_org"

p = Lark(
'''
start: rules

rules: (rule? _NL)* rule

rule: regex_body_rule | describe_rule | test_rule | comment

regex_body_rule: "body" CNAME perl_regex
describe_rule: "describe" CNAME description
test_rule: "test" CNAME outcome description
comment: COMMENT

perl_regex: "/" regex "/" opt_modifier?
opt_modifier: "i"

regex: primitive
       | _seperated{regex, "|"}  -> regex_alternatives
       | "(" regex ")" -> regex_group
       | regex regex+ -> regex_sequence
       | regex "?" -> regex_optional
       | regex "*" -> regex_zero_or_more
       | regex "+" -> regex_one_or_more
       | regex "{" NUMBER "}" -> regex_n_exactly
       | regex "{" NUMBER "," "}" -> regex_n_at_least
       | regex "{" "," NUMBER "}" -> regex_n_at_most
       | regex "{" NUMBER "," NUMBER "}" -> regex_n_to_m

primitive: "[" /[^]]/+ "]" -> charclass
         | "^" -> bol
         | "$" -> eol
         | "." -> anychar

description: /.+/

outcome: TESTOK | TESTFAILED
REGEX: "/" /[^\/]+/ "/"
TESTOK: "ok"
TESTFAILED: "failed"

_seperated{x, sep}: x (sep x)*  // Define a sequence of 'x sep x sep x ...'

%import common.WORD
%import common.WS_INLINE
%import common.CNAME
%import common.NUMBER
%import common.SH_COMMENT -> COMMENT
%import common.NEWLINE -> _NL

%ignore WS_INLINE
%ignore COMMENT
'''
    )

# test_file = assasin_dir+"/20_advance_fee.cf"
regex_funcs = []
all_filenames = [i for i in glob.glob(f'{assasin_dir}/*.*')]
for f in all_filenames:
	print(f)
	file = open(f, mode='r', encoding = "ISO-8859-1")
	lines = file.readlines()
	for line in lines:
		try:
			tree = p.parse(line.split('\n')[0])
			tokens = [t for t in tree.scan_values(lambda v: isinstance(v, Token))]
			for t in tokens:
				if(t.type=='REGEX'):
					regex_desc = t.value.strip('//i')
					re_obj = re.compile(regex_desc)
					regex_func = make_regex_lf(re_obj, name=f'regex_{regex_desc}')
					regex_funcs.append(regex_func)
		except Exception as e:
			continue


print(regex_funcs)
print(len(regex_funcs))

