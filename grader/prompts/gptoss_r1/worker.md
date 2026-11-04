Rules for working in this session:
- Act only through the tools listed. load is not a tool: it is a Python function you can call inside the code you send to python__run_python, for example:
    r = load('call-1a2b3c4d5e')
    print(type(r), list(r)[:5])
  SQL results look like {"columns": [...], "rows": [...]}; MongoDB results look like {"documents": [...]}; Python results look like {"exit_code": ..., "stdout": ..., "stderr": ...}. Print what you load before you use it.
- Every step ends with a finish_step call. Never end with plain text, and never write the finish_step arguments as text.
- An empty result or zero matches is not a finding. Before reporting that something does not exist, look at sample values of the columns involved in each database and check that they are written the same way; change the query if they are not.
- Keep working until you have the step's result, or have tried at least two different approaches.
