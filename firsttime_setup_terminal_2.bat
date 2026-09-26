# Start the RustFS S3 server using Docker
# web UI http://localhost:9001
# Test command: aws --endpoint-url http://localhost:9000 s3 ls
mkdir C:\rustfs
cd C:\rustfs
mkdir data
mkdir logs
docker run -d -p 9000:9000 -p 9001:9001 -v "%cd%\data:/data" -v "%cd%\logs:/logs" rustfs/rustfs:latest

# Use RustFS web UI to create access keys for S3 access. The default access key and secret key are "rustfsadmin" and "rustfsadmin". You can also use the following commands to set up the S3 bucket and upload sample logs to it:
# ATTENTION: Make sure to replace the access key and secret key with the ones you created in the RustFS web UI if you choose to use different keys.
# Default keys are assigned below for testing purposes only. You can also use the following commands to set up the S3 bucket and upload sample logs to it:
set AWS_ACCESS_KEY_ID=rustfsadmin
set AWS_SECRET_ACCESS_KEY=rustfsadmin
set AWS_DEFAULT_REGION=us-east-1
aws --endpoint-url http://localhost:9000 s3 mb s3://loghawk-data
aws --endpoint-url http://localhost:9000 s3 cp C:\loghawk_sample_logs.json s3://loghawk-data/raw/year=2026/month=09/day=23/sample_logs.json
aws --endpoint-url http://localhost:9000 s3 ls s3://loghawk-data

# If you choose to use SeaweedFS S3 server instead of RustFS, you can use the following command to start it using Docker Compose:
# Start the SeaweedFS S3 server using Docker Compose
# replace with current file name
# web UI http://localhost:8888
# Test command: aws --endpoint-url http://localhost:8333 s3 ls
# docker-compose -f src\loghawk\admin\seaweedfss3-docker-compose2.yml up

