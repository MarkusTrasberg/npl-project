def deploy():
    """Run deployment tasks."""
    from app import create_app

    app = create_app()
    app.app_context().push()

deploy()