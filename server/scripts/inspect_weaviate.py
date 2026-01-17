import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.auth import AuthApiKey
def main():
    # Correct for local Weaviate
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,  # ← HTTP, not HTTPS
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,  # ← No TLS
	auth_credentials=AuthApiKey("F3z3z27ZUls3pIQ5tW"),
    )

    print(client.is_ready())
    client.close()

if __name__ == "__main__":
    main()
