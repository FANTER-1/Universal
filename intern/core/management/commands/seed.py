from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from core.models import Role, Permission
from core.utils import create_permission, add_permission_to_role, assign_role

User = get_user_model()


def handle(self, *args, **kwargs):
    if Role.objects.exists() or Permission.objects.exists():
        self.stdout.write(self.style.WARNING('⚠️ База данных уже заполнена. Пропускаем'))
        return


class Command(BaseCommand):
    help = 'Seeds the database with initial data'

    def handle(self, *args, **kwargs):
        admin_role, _ = Role.objects.get_or_create(
            name='admin',
            defaults={'description': 'System administrator'}
        )

        user_role, _ = Role.objects.get_or_create(
            name='user',
            defaults={'description': 'Regular user'}
        )

        permissions_data = [
            ('user', 'view', 'View user accounts'),
            ('user', 'manage', 'Manage user accounts'),
            ('role', 'admin', 'Manage roles'),
            ('permission', 'admin', 'Manage permissions'),
            ('project', 'view', 'View projects'),
            ('project', 'create', 'Create projects'),
            ('project', 'edit', 'Edit projects'),
            ('project', 'delete', 'Delete projects'),
        ]

        for resource, action, desc in permissions_data:
            perm = create_permission(resource, action, desc)
            if perm:
                add_permission_to_role('admin', perm)

                if resource == 'project' and action == 'view':
                    add_permission_to_role('user', perm)

        admin_user, created = User.objects.get_or_create(
            email='админ',
            defaults={
                'first_name': 'Admin',
                'last_name': 'User'
            }
        )
        if created:
            admin_user.set_password('adminpassword')
            admin_user.is_superuser = True
            admin_user.save()

        assign_role(admin_user, 'admin')

        regular_user, created = User.objects.get_or_create(
            email='неадмин',
            defaults={
                'first_name': 'Анна',
                'last_name': 'EffectiveMobile'
            }
        )
        if created:
            regular_user.set_password('неадмин')
            regular_user.save()

        assign_role(regular_user, 'user')

        self.stdout.write(self.style.SUCCESS('✅ Database seeded successfully'))