import os
import weaviate
from weaviate.classes.init import Auth

def main():
    client = weaviate.connect_to_custom(
        http_host=os.environ["WEAVIATE_HTTP_HOST"],
        http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "443")),
        http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "true").lower() == "true",
        grpc_host=os.environ["WEAVIATE_GRPC_HOST"],
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "443")),
        grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "true").lower() == "true",
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),  # <-- important
    )

    print(client.is_ready())
    client.close()

if __name__ == "__main__":
    main()
