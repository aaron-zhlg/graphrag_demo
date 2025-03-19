from llm.deepseek_client import chat_with_deepseek
from gremlin_python.structure.graph import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
import boto3
import os


def extract_entities_relations_claims(post_data):

    json_format = """
        {"Entities": [{'entity_name': xxx, 'entity_type': xxx}, {'entity_name': xxx, 'entity_type': xxx}, ], 
        'Relations': [{'entity1': xxx, 'relation_type': xxx, 'entity2': xxx}, {'entity1': xxx, 'relation_type': xxx, 'entity2': xxx},],
        'Claims': [Claim1, Claim2, Claim3, Claim4, Claim5, Claim6],
        }
    """
    prompt = f"""
        Please analyze the following Reddit post data, and extract entities, relations, and claims.
        
        Reddit post data: [{post_data}]
        
        Extraction result should be in json format with 3 keys: Entities, Relations and Claims.
        {json_format}
        
        Please note:
        - Entity types can be: User, Subreddit, Post, Topic, Organization, Location, etc.
        - Relation types can be: PostedIn, BelongsTo, Mentions, Comments, AgreesWith, DisagreesWith, etc.
        - Claims are the opinions or assertions expressed in the post.
        - 'Do not wrap the json codes in JSON markers'
        """

    messages = [
        {
            'role': 'system', 'content': prompt,
        },
    ]
    try:
        response = chat_with_deepseek(messages)
        entities = response.get('Entities', [])
        relations = response.get('Relations', [])
        claims = response.get('Claims', [])
        return entities, relations, claims

    except Exception as e:
        print(f"Error processing post: {e}")
        return [], [], []


def save_to_neptune(post_id, entities, relations, claims, cluster_identifier, region_name="us-east-1"):
    """
    Saves extracted entities, relations, and claims to AWS Neptune using boto3 to get the endpoint.
    """
    neptune = boto3.client('neptune',
                           region_name=region_name,
                           aws_access_key_id=os.getenv('NEPTUNE_ACCESS_KEY'),
                           aws_secret_access_key=os.getenv('NEPTUNE_ACCESS_SECRET'))

    remote_connection = None  # Initialize remote_connection

    try:
        # Get the Neptune cluster endpoint
        response = neptune.describe_db_clusters(DBClusterIdentifier=cluster_identifier)
        endpoint = response['DBClusters'][0]['Endpoint']
        print(f"Endpoint: {endpoint}")
        neptune_endpoint = f"wss://{endpoint}:8182/gremlin"
        print(neptune_endpoint)

        graph = Graph()
        remote_connection = DriverRemoteConnection(neptune_endpoint, 'g')
        g = graph.traversal().withRemote(remote_connection)

        # Add Post node
        g.addV('Post').property('id', post_id).next()

        # Add Entities
        for entity in entities:
            g.addV(entity['entity_type']).property('name', entity['entity_name']).next()
            g.V(post_id).addE('mentions').to(g.V().has('name', entity['entity_name'])).next()

        # Add Relations
        for relation in relations:
            g.V().has('name', relation['entity1']).as_('a').V().has('name', relation['entity2']).addE(relation['relation_type']).from_('a').next()

        # Add Claims (as Post node properties)
        for i, claim in enumerate(claims):
            g.V(post_id).property(f'claim_{i+1}', claim).next()

    except Exception as e:
        print(f"Error saving to Neptune: {e}")

    finally:
        if remote_connection:
            try:
                remote_connection.close()
                if hasattr(remote_connection, '_client') and hasattr(remote_connection._client, '_client_session'):
                    remote_connection._client._client_session.close()
            except Exception as close_error:
                print(f"Error closing connection: {close_error}")

cluster_identifier = 'test'


def process_post(post_id):
    post_data = ('10-k report on Nvidia read. What company to look at next? I am a long term investor, '
                 'that wants to buy high quality companies, with wide moats, strong financial metrics, great future outlook, '
                 'pricing power, high margins and revenue growth. I am invested in a few of them, but I am looking for your opinions based')
    entities, relations, claims = extract_entities_relations_claims(post_data)
    print(entities)
    print(relations)
    print(claims)
    save_to_neptune(post_id, entities, relations, claims, cluster_identifier)
