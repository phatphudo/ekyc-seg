import sys
import pprint
from model import Trainer
from tool import Config

def main():
    config = Config.load_yaml('config.yml')
    print(config)
    
    trainer = Trainer(config)
    
    print(f"Current run: [{trainer.run_name}] on device [{trainer.device}]")
    confirm = input("Please confirm (y/n): ")
    if confirm == 'y':
        trainer.fit()
        trainer.eval()
    elif confirm == 'n':
        sys.exit()


if __name__ == '__main__':
    main()