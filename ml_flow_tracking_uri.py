from zenml.client import Client
from uuid import UUID

root_path = Client().active_stack.artifact_store.path
print(root_path)