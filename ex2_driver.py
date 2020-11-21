import ex2


spc = ex2.Spell_Checker()
error_tables = spc.learn_error_tables('commmon_errors.txt')
print('{')
for label in ['insertion', 'deletion', 'substitution', 'transposition']:
    print(f'\'{label}\': {error_tables[label]}')
print('}')
