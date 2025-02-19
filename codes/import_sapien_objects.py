import sapien
from backports import tarfile

token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImthcHBlbEBpc2lyLnVwbWMuZnIiLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE3Mzk3OTM0NTQsImV4cCI6MTc3MTMyOTQ1NH0.SQK9LUKZzLTVNCh4QzRG_3HaMOXNajJ1RSSgjRKJ8-0"
for i in range(10000,20000):
    print(i)
    try:
        urdf_file = sapien.asset.download_partnet_mobility(i, token)
    except Exception as e:
        print(i, " does not exist")
        pass
# create scene and URDF loader
