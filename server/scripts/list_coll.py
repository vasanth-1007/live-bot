import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.auth import AuthApiKey

def main():
    # 1. Connect to local Weaviate
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,  # HTTP
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,  # No TLS
        auth_credentials=AuthApiKey("F3z3z27ZUls3pIQ5tW"),
    )

    try:
        # Check connection first
        if client.is_ready():
            print("Successfully connected to Weaviate!")
            
            # 2. List all collections
            # client.collections.list_all() returns a dictionary where keys are collection names
            collections = client.collections.list_all()

            if collections:
                print(f"\nFound {len(collections)} collection(s):")
                print("-" * 30)
                for name in collections.keys():
                    print(f"- {name}")
            else:
                print("\nNo collections found in this Weaviate instance.")
        else:
            print("Weaviate is not ready.")
            
    finally:
        # 3. Always close the connection
        client.close()

if __name__ == "__main__":
    main()
