# Generated by Django 5.0.6 on 2024-06-07 16:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0004_respuesta_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='respuesta',
            name='quizz_id',
            field=models.IntegerField(default=23),
            preserve_default=False,
        ),
    ]