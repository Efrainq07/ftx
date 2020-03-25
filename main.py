from rest.client import FtxClient
from secret.keys import api_key,api_secret

client = FtxClient(api_key = api_key,api_secret= api_secret,subaccount_name='autotrade')
#historical = client.get_window('XTZBULL/USD',15,'day')
