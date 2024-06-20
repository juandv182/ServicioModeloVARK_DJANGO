#!/usr/bin/env bash
# Exit on error
set -o errexit

# Modify this line as needed for your package manager (pip, poetry, etc.)
pip install -r requirements.txt

# Convert static asset files
python manage.py collectstatic --no-input

python manage.py makemigrations --empty auth
python manage.py makemigrations --empty admin
python manage.py makemigrations --empty contenttypes
python manage.py makemigrations --empty sessions
python manage.py makemigrations --empty messages
python manage.py makemigrations --empty staticfiles
python manage.py makemigrations staticfiles
# Apply any outstanding database migrations
python manage.py migrate classifier