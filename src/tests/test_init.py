import pytest
from unittest.mock import patch, MagicMock
import init
import os
import subprocess

def test_ask_recreate_image_yes(mocker):
    mocker.patch('builtins.input', return_value='Y')
    assert init.ask_recreate_image() is True

def test_ask_recreate_image_no(mocker):
    mocker.patch('builtins.input', return_value='N')
    assert init.ask_recreate_image() is False

def test_set_docker_username(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch('builtins.input', return_value='docker_user')
    docker_username = init.set_docker_username()
    assert docker_username == 'docker_user'
    assert os.environ['YOUR_DOCKER_USERNAME'] == 'docker_user'

def test_run_command_success(mocker):
    mock_subproc_popen = mocker.patch('subprocess.Popen')
    process_mock = MagicMock()
    attrs = {'communicate.return_value': (b'output', b''), 'returncode': 0}
    process_mock.configure_mock(**attrs)
    mock_subproc_popen.return_value = process_mock

    output = init.run_command("echo test")
    assert output == 'output'

def test_run_command_failure(mocker):
    mock_subproc_popen = mocker.patch('subprocess.Popen')
    process_mock = MagicMock()
    attrs = {'communicate.return_value': (b'', b'error'), 'returncode': 1}
    process_mock.configure_mock(**attrs)
    mock_subproc_popen.return_value = process_mock

    with pytest.raises(SystemExit):
        init.run_command("false")

def test_ask_yes_no_yes(mocker):
    mocker.patch('builtins.input', return_value='yes')
    assert init.ask_yes_no("Test question?") is True

def test_ask_yes_no_no(mocker):
    mocker.patch('builtins.input', return_value='no')
    assert init.ask_yes_no("Test question?") is False

def test_setup_and_push_docker_image(mocker):
    mock_ask_yes_no = mocker.patch('init.ask_yes_no', return_value=True)
    mock_run_command = mocker.patch('init.run_command')
    docker_username = "docker_user"

    init.setup_and_push_docker_image(docker_username)

    mock_run_command.assert_any_call("docker build -t stock-prediction-app .")
    mock_run_command.assert_any_call(f"docker tag stock-prediction-app:latest {docker_username}/stock-prediction-app:latest")
    mock_run_command.assert_any_call("docker login")
    mock_run_command.assert_any_call(f"docker push {docker_username}/stock-prediction-app:latest")

def test_deploy_to_kubernetes(mocker):
    mock_ask_yes_no = mocker.patch('init.ask_yes_no', return_value=True)
    mock_run_command = mocker.patch('init.run_command')
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='yaml_content'))
    docker_username = "docker_user"

    init.deploy_to_kubernetes(docker_username)

    mock_open.assert_any_call('kubernetes-deployment.yaml', 'r')
    mock_open.assert_any_call('temp_kube_config.yaml', 'w')
    mock_run_command.assert_called_with("kubectl apply -f temp_kube_config.yaml")
