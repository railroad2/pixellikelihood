from configparser import ConfigParser


def read_config(fname):
    parser = ConfigParser()
    parser.read(fname)

    config = {} 
    for sect in parser.sections():
        for opt in parser[sect]:
            config[opt] = string2config(parser.get(sect, opt))
            print (f'{opt} : {config[opt]}')
    
    return config


def get_config(cfgdict, name, default=None):
    try:
        val = cfgdict[name]
    except KeyError:
        val = default

    return val


def string2config(s):
    c = None
    try:
        c = eval(s)
    except:
        c = s

    return c
        
