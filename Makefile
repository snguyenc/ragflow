docb-base: 
	docker build -f Dockerfile.deps -t infiniflow/ragflow_deps .
docp:
	docker push  reg.securityzone.vn/sonnguyen/ragflow:book-1.0
docb:
	docker build --platform=linux/amd64 --build-arg LIGHTEN=1 -f Dockerfile -t reg.securityzone.vn/sonnguyen/ragflow:book-1.0 .