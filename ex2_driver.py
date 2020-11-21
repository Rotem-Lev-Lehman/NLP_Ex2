import ex2


spc = ex2.Spell_Checker()
error_tables = spc.learn_error_tables('commmon_errors.txt')
print(error_tables)
