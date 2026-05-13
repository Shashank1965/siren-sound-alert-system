from django.db import models

# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'

class ImageModel(models.Model):
    username = models.CharField(max_length=150)
    text_description = models.CharField(max_length=1000)
    date_now = models.DateTimeField(auto_now=True)
    image_generated = models.ImageField() 

    def __str__(self):
        return self.username
    
    class Meta:
        db_table = 'image_model'

