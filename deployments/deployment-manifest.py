from jinja2 import Template
import os 
import sys

root_directory = '/blockchain/brownie/deployments'

def generate_save_directory(directory, file):
    print(directory, file)
    generate_directory = os.path.join(root_directory, directory)
    print(generate_directory)
    if not os.path.exists(generate_directory):
        os.makedirs(directory)
    return os.path.join(generate_directory, file)

def generate_pvc(generate_directory, namespace, storage_class_name, version_name):
    with open(generate_save_directory('templates', '01_persistent_volume_claim.yml'), 'r') as f:
        yml = Template(f.read())
        yml = yml.render(namespace=namespace, storage_class_name=storage_class_name, version_name=version_name)
    print('writing kubernetes configuration to', generate_directory)
    with open(generate_directory, 'w+') as f:
        f.write(yml)

def generate_deployment(generate_directory, namespace, environment, web3_infura_project_id, web3_infura_project_secret, web3_infura_mainnet_wss, web3_infura_mainnet_https, web3_infura_kovan_wss, web3_infura_kovan_https, wallet_private_key_mainnet, wallet_private_key_kovan, version_name):
    with open(generate_save_directory('templates', '02_deployments.yml')) as f:
        yml = Template(f.read())
        yml = yml.render(namespace=namespace, environment=environment, web3_infura_project_id=web3_infura_project_id, web3_infura_project_secret=web3_infura_project_secret, web3_infura_mainnet_wss=web3_infura_mainnet_wss, web3_infura_mainnet_https=web3_infura_mainnet_https, web3_infura_kovan_wss=web3_infura_kovan_wss, web3_infura_kovan_https=web3_infura_kovan_https, wallet_private_key_mainnet=wallet_private_key_mainnet, wallet_private_key_kovan=wallet_private_key_kovan, version_name=version_name)
    print('writing kubernetes configuration to', generate_directory)
    with open(generate_directory, 'w+') as f:
        f.write(yml)

def main():
    _01 = generate_save_directory(sys.argv[1], '01_persistent_volume_claim.yml')
    generate_pvc(generate_directory=_01, namespace=sys.argv[2], storage_class_name=sys.argv[3], version_name=sys.argv[13])

    _02 = generate_save_directory(sys.argv[1], '02_deployments.yml')
    generate_deployment(generate_directory=_02, namespace=sys.argv[2], environment=sys.argv[4], web3_infura_project_id=sys.argv[5], web3_infura_project_secret=sys.argv[6], web3_infura_mainnet_wss=sys.argv[7], web3_infura_mainnet_https=sys.argv[8], web3_infura_kovan_wss=sys.argv[9], web3_infura_kovan_https=sys.argv[10], wallet_private_key_mainnet=sys.argv[11], wallet_private_key_kovan=sys.argv[12], version_name=sys.argv[13])

# sudo python3 manifest.py test default namespace environment web3_infura_project_id web3_infura_project_secret web3_infura_mainnet_wss web3_infura_mainnet_https web3_infura_kovan_wss web3_infura_kovan_https wallet_private_key_mainnet wallet_private_key_kovan version_name mainnet
if __name__ == '__main__':
    main()
