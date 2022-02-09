commit:
	# Clears all cell outputs in exercises
	jupyter nbconvert --clear-output --inplace exercises/*/*.ipynb

	# Lints
	black exercises/
	isort exercises/

	# Checks linting
	flake8 exercises/

	git add .
	@read -p "Commit message: " comMsg; \
	git commit -m "$$comMsg"