# Generated by Django 5.0.6 on 2024-06-07 16:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0005_respuesta_quizz_id'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='respuesta',
            name='user',
        ),
    ]