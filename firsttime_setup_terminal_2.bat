# Start the RustFS S3 server using Docker
# web UI http://localhost:9001
# Test command: aws --endpoint-url http://localhost:9000 s3 ls
mkdir C:\rustfs
cd C:\rustfs
mkdir data
mkdir logs
docker run -d -p 9000:9000 -p 9001:9001 -v "%cd%\data:/data" -v "%cd%\logs:/logs" rustfs/rustfs:latest

# If you choose to use SeaweedFS S3 server instead of RustFS, you can use the following command to start it using Docker Compose:
# Start the SeaweedFS S3 server using Docker Compose
# replace with current file name
# web UI http://localhost:8888
# Test command: aws --endpoint-url http://localhost:8333 s3 ls
# docker-compose -f src\loghawk\admin\seaweedfss3-docker-compose.yml up

