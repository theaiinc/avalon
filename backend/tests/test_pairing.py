import pytest

import pairing


class MemoryKeyring:
    class errors:
        class PasswordDeleteError(Exception):
            pass

    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, value):
        self.values[(service, username)] = value

    def delete_password(self, service, username):
        self.values.pop((service, username), None)


@pytest.fixture
def isolated_keyring(monkeypatch):
    store = MemoryKeyring()
    monkeypatch.setattr(pairing, "keyring", store)
    pairing.reset_for_tests()
    yield store
    pairing.reset_for_tests()


def test_pairing_code_is_single_use_and_creates_peer_credential(isolated_keyring):
    code = pairing.create_pairing_code()
    peer = pairing.device_info()
    signature = pairing.sign_pairing("", code["code"])

    result = pairing.accept_pairing("", code["code"], "other-device", peer["public_key"], signature)

    assert result["peer"]["name"] == "other-device"
    assert pairing.validate_peer_token(result["api_key"]) is True
    with pytest.raises(ValueError, match="invalid or expired"):
        pairing.accept_pairing("", code["code"], "replay", peer["public_key"], signature)


def test_expired_pairing_code_is_rejected(monkeypatch, isolated_keyring):
    now = [1_000_000]
    monkeypatch.setattr(pairing.time, "time", lambda: now[0])
    code = pairing.create_pairing_code()
    peer = pairing.device_info()
    signature = pairing.sign_pairing(code["session_id"], code["code"])
    now[0] += pairing.PAIRING_TTL_SECONDS + 1

    with pytest.raises(ValueError, match="invalid or expired"):
        pairing.accept_pairing(code["session_id"], code["code"], "late", peer["public_key"], signature)


def test_wrong_code_does_not_consume_pairing_session(isolated_keyring):
    code = pairing.create_pairing_code()
    peer = pairing.device_info()
    signature = pairing.sign_pairing(code["session_id"], code["code"])

    with pytest.raises(ValueError, match="invalid or expired"):
        pairing.accept_pairing(code["session_id"], "WRONG234", "wrong", peer["public_key"], signature)
    assert pairing.accept_pairing(
        code["session_id"], code["code"], "correct", peer["public_key"], signature
    )["peer"]["name"] == "correct"
