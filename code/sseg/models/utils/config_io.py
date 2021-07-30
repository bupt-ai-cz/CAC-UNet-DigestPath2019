from yaml import load

def load_config(config_path):
    with open(config_path, 'rb') as f:
        cont = f.read()
    cf = load(cont)
    return cf



if __name__ == "__main__":
    cf = load_config("../default.yaml")
    print(cf)
    print(cf.MODEL)