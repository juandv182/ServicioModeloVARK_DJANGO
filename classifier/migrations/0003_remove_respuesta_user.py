# Generated by Django 5.0.6 on 2024-06-07 15:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0002_remove_respuesta_pregunta_remove_respuesta_usuario_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='respuesta',
            name='user',
        ),
    ]