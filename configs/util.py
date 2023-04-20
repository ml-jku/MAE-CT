def cliarg_or_staticvalue(cliarg, staticvalue):
    if cliarg is not None:
        return cliarg
    return staticvalue
