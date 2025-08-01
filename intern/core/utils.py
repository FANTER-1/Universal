from django.core.exceptions import ObjectDoesNotExist
from .models import Role, Permission, UserRole, RolePermission


def assign_role(user, role_name):
    try:
        role = Role.objects.get(name=role_name)
        UserRole.objects.get_or_create(user=user, role=role)
        return True
    except ObjectDoesNotExist:
        return False


def create_permission(resource, action, description=None):
    permission, created = Permission.objects.get_or_create(
        resource_name=resource,
        action=action,
        defaults={'description': description}
    )
    return permission


def add_permission_to_role(role_name, permission):
    try:
        role = Role.objects.get(name=role_name)
        if isinstance(permission, str):
            # Если передана строка формата "resource.action"
            resource, action = permission.split('.')
            permission = Permission.objects.get(resource_name=resource, action=action)

        RolePermission.objects.get_or_create(role=role, permission=permission)
        return True
    except (ObjectDoesNotExist, ValueError):
        return False