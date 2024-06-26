# Generated by Django 5.0.6 on 2024-05-18 16:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='respuesta',
            name='pregunta',
        ),
        migrations.RemoveField(
            model_name='respuesta',
            name='usuario',
        ),
        migrations.RemoveField(
            model_name='respuesta',
            name='respuesta',
        ),
        migrations.AddField(
            model_name='respuesta',
            name='age',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='gender',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='learn',
            field=models.CharField(default='v', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q1',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q10',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q11',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q12',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q13',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q14',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q15',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q2',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q3',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q4',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q5',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q6',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q7',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q8',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='q9',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='respuesta',
            name='user',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
        migrations.DeleteModel(
            name='Pregunta',
        ),
        migrations.DeleteModel(
            name='Usuario',
        ),
    ]
