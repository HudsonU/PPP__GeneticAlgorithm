from pulp.apis import PULP_CBC_CMD
solver = PULP_CBC_CMD(msg=True)
print(solver.path)   # path to the CBC executable
