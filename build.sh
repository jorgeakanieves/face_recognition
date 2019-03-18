 docker build -t face-recognition -f Dockerfile .
 docker run --rm -t -p 8889:8888 --name face-recognition face-recognition
