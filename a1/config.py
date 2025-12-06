"""
A1 Configuration: AWS Braket Settings Loader

This module loads configuration from environment variables or .env file.

Usage:
    from a1.config import config
    
    print(config.aws_region)
    print(config.s3_bucket)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


def load_dotenv(env_path: Optional[Path] = None) -> dict:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks in current dir and a1/
        
    Returns:
        Dict of loaded environment variables
    """
    loaded = {}
    
    # Find .env file
    if env_path is None:
        candidates = [
            Path.cwd() / '.env',
            Path(__file__).parent / '.env',
            Path.cwd() / 'a1' / '.env',
        ]
        for candidate in candidates:
            if candidate.exists():
                env_path = candidate
                break
    
    if env_path is None or not env_path.exists():
        return loaded
    
    # Parse .env file
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                loaded[key] = value
                # Also set in os.environ
                os.environ.setdefault(key, value)
    
    return loaded


@dataclass
class BraketConfig:
    """AWS Braket configuration settings."""
    
    # AWS credentials
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_profile: Optional[str] = None
    aws_region: str = 'us-east-1'
    
    # S3 settings
    s3_bucket: Optional[str] = None
    s3_prefix: str = 'a1-experiments'
    
    # Device ARNs
    simulator_sv1: str = 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
    simulator_tn1: str = 'arn:aws:braket:::device/quantum-simulator/amazon/tn1'
    simulator_dm1: str = 'arn:aws:braket:::device/quantum-simulator/amazon/dm1'
    device_ionq: str = 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
    device_rigetti: str = 'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3'
    device_iqm: str = 'arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet'
    
    # Experiment settings
    default_shots_simulator: int = 1000
    default_shots_qpu: int = 100
    num_trials: int = 5
    task_timeout: int = 3600
    
    # Cost control
    enable_qpu_execution: bool = False
    max_cost_per_experiment: float = 10.00
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'a1_braket.log'
    
    @classmethod
    def from_env(cls) -> 'BraketConfig':
        """Create config from environment variables."""
        # Load .env file if exists
        load_dotenv()
        
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, str(default)).lower()
            return val in ('true', '1', 'yes', 'on')
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except ValueError:
                return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except ValueError:
                return default
        
        return cls(
            # AWS credentials
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_profile=os.environ.get('AWS_PROFILE'),
            aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
            
            # S3 settings
            s3_bucket=os.environ.get('BRAKET_S3_BUCKET'),
            s3_prefix=os.environ.get('BRAKET_S3_PREFIX', 'a1-experiments'),
            
            # Device ARNs
            simulator_sv1=os.environ.get(
                'BRAKET_SIMULATOR_SV1',
                'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
            ),
            simulator_tn1=os.environ.get(
                'BRAKET_SIMULATOR_TN1',
                'arn:aws:braket:::device/quantum-simulator/amazon/tn1'
            ),
            simulator_dm1=os.environ.get(
                'BRAKET_SIMULATOR_DM1',
                'arn:aws:braket:::device/quantum-simulator/amazon/dm1'
            ),
            device_ionq=os.environ.get(
                'BRAKET_DEVICE_IONQ',
                'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
            ),
            device_rigetti=os.environ.get(
                'BRAKET_DEVICE_RIGETTI',
                'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3'
            ),
            device_iqm=os.environ.get(
                'BRAKET_DEVICE_IQM',
                'arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet'
            ),
            
            # Experiment settings
            default_shots_simulator=get_int('DEFAULT_SHOTS_SIMULATOR', 1000),
            default_shots_qpu=get_int('DEFAULT_SHOTS_QPU', 100),
            num_trials=get_int('NUM_TRIALS', 5),
            task_timeout=get_int('TASK_TIMEOUT', 3600),
            
            # Cost control
            enable_qpu_execution=get_bool('ENABLE_QPU_EXECUTION', False),
            max_cost_per_experiment=get_float('MAX_COST_PER_EXPERIMENT', 10.00),
            
            # Logging
            log_level=os.environ.get('LOG_LEVEL', 'INFO'),
            log_file=os.environ.get('LOG_FILE', 'a1_braket.log'),
        )
    
    @property
    def s3_location(self) -> Optional[str]:
        """Get S3 location for Braket results."""
        if self.s3_bucket:
            return f's3://{self.s3_bucket}/{self.s3_prefix}'
        return None
    
    def get_device_arn(self, device_name: str) -> str:
        """
        Get device ARN by name.
        
        Args:
            device_name: One of 'sv1', 'tn1', 'dm1', 'ionq', 'rigetti', 'iqm'
            
        Returns:
            Device ARN string
        """
        devices = {
            'sv1': self.simulator_sv1,
            'tn1': self.simulator_tn1,
            'dm1': self.simulator_dm1,
            'ionq': self.device_ionq,
            'rigetti': self.device_rigetti,
            'iqm': self.device_iqm,
        }
        return devices.get(device_name.lower(), self.simulator_sv1)
    
    def is_simulator(self, device_name: str) -> bool:
        """Check if device is a simulator."""
        return device_name.lower() in ('sv1', 'tn1', 'dm1', 'local')
    
    def validate(self) -> list:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check S3 bucket for cloud execution
        if not self.s3_bucket:
            errors.append("BRAKET_S3_BUCKET is not set (required for AWS execution)")
        
        # Check credentials
        has_direct_creds = self.aws_access_key_id and self.aws_secret_access_key
        has_profile = self.aws_profile
        if not has_direct_creds and not has_profile:
            errors.append("No AWS credentials configured (set AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)")
        
        return errors
    
    def print_status(self):
        """Print configuration status."""
        print("=" * 60)
        print("A1 Braket Configuration Status")
        print("=" * 60)
        print(f"Region: {self.aws_region}")
        print(f"S3 Bucket: {self.s3_bucket or '(not set)'}")
        print(f"Profile: {self.aws_profile or '(not set)'}")
        print(f"Direct Credentials: {'Yes' if self.aws_access_key_id else 'No'}")
        print(f"QPU Execution Enabled: {self.enable_qpu_execution}")
        print(f"Default Shots (Simulator): {self.default_shots_simulator}")
        print(f"Default Shots (QPU): {self.default_shots_qpu}")
        print()
        
        errors = self.validate()
        if errors:
            print("⚠️  Configuration Issues:")
            for err in errors:
                print(f"   - {err}")
        else:
            print("✅ Configuration valid")
        print("=" * 60)


# Global config instance
config = BraketConfig.from_env()


if __name__ == "__main__":
    config.print_status()

