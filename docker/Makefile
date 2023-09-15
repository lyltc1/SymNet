build-cuda116-torch112-bop:
	docker build -f Dockerfile \
		-t symnet/env:cuda116-torch112-detectron2-bop-0.0.6 \
		--network=host\
		--build-arg http_proxy=http://127.0.0.1:7890 \
    	--build-arg https_proxy=http://127.0.0.1:7890 \
    	--build-arg "NO_PROXY=localhost,127.0.0.1,.example.com" \
		.
