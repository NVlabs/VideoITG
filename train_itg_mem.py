try:
    import transformer_engine
    import transformer_engine_extensions
except:
    print("having trouble importing transformer-engine!")
    
from train_itg import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
