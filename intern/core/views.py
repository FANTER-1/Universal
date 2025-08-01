from datetime import timezone
from venv import logger

from pydantic import ValidationError
from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate, logout
from django.shortcuts import get_object_or_404
from .models import User, Role, Permission
from .serializers import UserSerializer, RoleSerializer, PermissionSerializer
from .permissions import HasPermission


class AuthViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['post'])
    def register(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({'token': token.key}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def login(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        if not email or not password:
            return Response({'error': 'Email and password required'}, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(email=email, password=password)

        if user:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({'token': token.key})

        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

    @action(detail=False, methods=['post'])
    def logout(self, request):
        if request.user.is_authenticated:
            Token.objects.filter(user=request.user).delete()
            logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.filter(is_active=True)
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, HasPermission]
    permission_resource = 'user'
    permission_action = 'manage'

    def get_permission_action(self):
        if self.action in ['list', 'retrieve']:
            return 'view'
        return self.action

    def perform_destroy(self, instance):
        instance.is_active = False
        instance.deleted_at = timezone.now()
        instance.save()


class RoleViewSet(viewsets.ModelViewSet):
    queryset = Role.objects.all()
    serializer_class = RoleSerializer
    permission_classes = [permissions.IsAuthenticated, HasPermission]
    permission_resource = 'role'
    permission_action = 'admin'


class PermissionViewSet(viewsets.ModelViewSet):
    queryset = Permission.objects.all()
    serializer_class = PermissionSerializer
    permission_classes = [permissions.IsAuthenticated, HasPermission]
    permission_resource = 'permission'
    permission_action = 'admin'


class ProjectViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated, HasPermission]
    permission_resource = 'project'

    def get_permission_action(self):
        return self.action_map.get(self.action, self.action)

    def list(self, request):
        if not request.user.has_perm('project.view'):
            return Response(status=status.HTTP_403_FORBIDDEN)

        return Response([
            {"id": 1, "name": "Project Alpha"},
            {"id": 2, "name": "Project Beta"}
        ])

    def create(self, request):
        try:
            if not request.user.has_perm('project.create'):
                return Response(status=status.HTTP_403_FORBIDDEN)

            name = request.data.get('name')
            if not name:
                raise ValidationError("Project name is required")

            project = {"id": 3, "name": name}

            return Response(project, status=status.HTTP_201_CREATED)

        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return Response(
                {"error": "Validation failed", "details": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )