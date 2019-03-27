docker build -t face-recognition -f Dockerfile .
docker run --rm -t -p 8889:8888 --name face-recognition face-recognition
# docker run --rm -t -p 8889:8888 -p 6006:6006 -v C:\dev\workspace\test-pocs\face-recognition:/face-recognition --name face-recognition face-recognition:1.0.0 bash
docker run --rm -t -p 8889:8888 -p 6006:6006 -v ./:/face-recognition --name face-recognition face-recognition bash
