# Generated by Django 5.0.6 on 2024-06-07 17:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0006_remove_respuesta_user'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='respuesta',
            table='learning_predictions',
        ),
    ]