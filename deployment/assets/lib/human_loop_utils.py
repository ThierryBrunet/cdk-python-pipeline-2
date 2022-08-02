import boto3
import json
import os
import time
from botocore.exceptions import ClientError



WORKFLOW_TABLE = os.environ['workflow_table_name']


def get_workflow_metadata(workflow_key):
    item = boto3.resource('dynamodb').Table(WORKFLOW_TABLE).get_item(
        Key=workflow_key
    )['Item']
    return item


def update_workflow_metadata(workflow_key, updates):
    boto3.resource('dynamodb').Table(WORKFLOW_TABLE).update_item(
        Key=workflow_key,
        AttributeUpdates={k: {'Value': v} for k,v in updates.items()}
    )
    
    
class Workflow:
    def __init__(self, workflow_table_name, workflow_id):
        self.table_name = workflow_table_name
        self.table = self._table()
        self.workflow_id = workflow_id
        self.workflow_key = {'workflow_id':workflow_id}
        self.data = self._get_workflow_metadata()
        
    def update(self, data):
        return self.table.update_item(
            Key=self.workflow_key,
            AttributeUpdates={k: {'Value': v} for k,v in data.items()}
        )

    def stop(self):
        """
        Stops the workflow (not reversible)
        Keeps workflow record but marks as complete
        Deletes queue and stops any loops in A2i workflow
        """
        self.update({
            'status': 'stopped'
        })
        self._queue().delete()
        self._clear_a2i_tasks()
    
    def _table(self):
        return boto3.resource('dynamodb').Table(self.table_name)
        
    def _queue(self):
        return boto3.resource('sqs').Queue(self.data['queue_url'])
    
    def _get_workflow_metadata(self):
        item = self.table.get_item(
            Key=self.workflow_key
        )['Item']
        return item

    def _clear_a2i_tasks(self):
        a2i = boto3.client('sagemaker-a2i-runtime')
        paginator = a2i.get_paginator('list_human_loops')
        response_iterator = paginator.paginate(
            FlowDefinitionArn=self.data['a2i_flow_arn']
        )
        for response in response_iterator:
            for task in response['HumanLoopSummaries']:
                if task['HumanLoopStatus']=='InProgress':
                    a2i.stop_human_loop(HumanLoopName=task['HumanLoopName'])
    
    def _generate_human_loop_name(self, item_id):
        """
        Generate name for human loop.
        Has format {workflow_id}--{timestamp}--{item_id}
        Truncated to maximum of 63 characters, not ending in a "-"
        """
        timestamp = str(int(time.time()))
        return f'{self.workflow_id}--{timestamp}--{item_id}'[:63].strip('-')
    
    def instantiate_human_loop(self, item):
        a2i = boto3.client('sagemaker-a2i-runtime')
        # human loop name follows convention {workflow_id}--{input id}--{timestamp}
        # human_loop_name = f'{workflow["workflow_id"]}--{item["reference"]["id"]}--{generate_timestamp()}'
        human_loop_name = self._generate_human_loop_name(item['reference']['id'])
        r = a2i.start_human_loop(
            HumanLoopName=human_loop_name,
            FlowDefinitionArn=self.data['a2i_flow_arn'],
            HumanLoopInput={
                "InputContent": json.dumps(item)
            }
        )
        return r
    
    def instantiate_loops_from_queue(self, num_items=1):
        for i in range(num_items):
            queue = boto3.resource('sqs').Queue(self.data['queue_url'])
            new_messages = queue.receive_messages(MaxNumberOfMessages=1)
            if len(new_messages)>0:
                message = new_messages[0]
                item = json.loads(message.body)
                print(f'Queuing new item: {item}')
                self.instantiate_human_loop(item)
                message.delete()
            else:
                print('No new items found in queue')
    
    def stop_pending_loops(self):
        a2i = boto3.client('sagemaker-a2i-runtime')
        paginator = a2i.get_paginator('list_human_loops')
        response_iterator = paginator.paginate(
            FlowDefinitionArn=self.data['a2i_flow_arn']
        )
        for response in response_iterator:
            for task in response['HumanLoopSummaries']:
                if task['HumanLoopStatus']=='InProgress':
                    a2i.stop_human_loop(HumanLoopName=task['HumanLoopName'])
    
    def queue_items(self, items_to_queue):
        """
        Queue items in batches of 10
        """
        queue = self._queue()
        # send items to queue
        for chunk in self._chunker(items_to_queue, 10):
            queue.send_messages(
                Entries=[
                    {'MessageBody':json.dumps(item), 'Id':str(i)} for i,item in enumerate(chunk)
                ]
            )
    
    def _iter_list_human_loops(self, **list_human_loops_params):
        a2i = boto3.client('sagemaker-a2i-runtime')
        paginator = a2i.get_paginator('list_human_loops')
        response_iterator = paginator.paginate(
            **list_human_loops_params
        )
        
        for response in response_iterator:
            for task in response['HumanLoopSummaries']:
                yield task
                
    def list_human_loops(self, **list_human_loops_params):
        return list(self._iter_list_human_loops(**list_human_loops_params))
                
    
    @staticmethod        
    def _chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def purge_queue(self):
        try:
            self._queue().purge()
        except ClientError as e:
            if e.response['Error']['Code']=='AWS.SimpleQueueService.PurgeQueueInProgress':
                return {'statusCode': 403, 'body': 'Purge queue in progress'}
            else:
                raise e
    

def query_ddb_paginated_iter(table, **query_kwargs):
    """
    Iterate over query results
    """
    done = False
    start_key = None
    while not done:
        if start_key:
            query_kwargs['ExclusiveStartKey'] = start_key
        response = table.query(**query_kwargs)
        for item in response['Items']:
            yield item
        start_key = response.get('LastEvaluatedKey', None)
        done = start_key is None


def get_workteam_arn(workteam_name):
    """
    Get workteam arn by workteam name
    """
    workteam_list = boto3.client('sagemaker').list_workteams(NameContains=workteam_name)['Workteams']
    for workteam in workteam_list:
        if workteam['WorkteamName'] == workteam_name:
            return workteam['WorkteamArn']
    raise Exception(f'Workteam named {workteam_name} not found')
