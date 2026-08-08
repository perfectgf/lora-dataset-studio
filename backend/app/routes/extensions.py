from flask import Blueprint, current_app, jsonify

bp = Blueprint('extensions', __name__, url_prefix='/api/extensions')


@bp.get('/')
def list_extensions():
    return jsonify({'extensions': current_app.config.get('EXTENSIONS_MANIFEST', [])})
