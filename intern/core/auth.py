from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions
from rest_framework.authtoken.models import Token

User = get_user_model()


class CustomTokenAuthentication(authentication.TokenAuthentication):
    keyword = 'Bearer'

    def authenticate_credentials(self, key):
        try:
            token = Token.objects.select_related('user').get(key=key)
            user = token.user

            if not user.is_active:
                raise exceptions.AuthenticationFailed('User inactive or deleted')

            return (user, token)
        except Token.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid token')


class CustomAuthBackend:
    def authenticate(self, request, email=None, password=None):
        try:
            user = User.objects.get(email=email)
            if not user.is_active:
                return None
            if user.check_password(password):
                return user
        except User.DoesNotExist:
            return None
        return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None