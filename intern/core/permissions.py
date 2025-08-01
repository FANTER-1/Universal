from rest_framework import permissions


class HasPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.user.is_superuser:
            return True

        resource = getattr(view, 'permission_resource', None)
        action = getattr(view, 'permission_action', None)

        if not resource or not action:
            return False

        permission_codename = f"{resource}.{action}"

        return request.user.has_perm(permission_codename)

    def has_object_permission(self, request, view, obj):
        return self.has_permission(request, view)
