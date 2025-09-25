# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import json
from typing import Dict, List, Literal, Optional, Tuple

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import (AssignmentInstruct, DistServeRegisterMRMessage, MigrationAssignment,
                                               MigrationExecutionBatch)
from lmdeploy.utils import get_logger

from ..config import CacheConfig, ModelConfig

logger = get_logger('lmdeploy')

# 定义图像特征块的形状
FEATURE_BLOCK_SHAPE = (256, 4096)

class EncoderCacheEngine:
    """Manages the memory pool for image features.

    This engine allocates and manages a contiguous block of GPU memory
    to store image embeddings transferred from an encoder. It is adapted for
    an encoder-LLM separated architecture.

    Args:
        cache_config (CacheConfig): Configuration for the cache, such as the
            number of blocks.
        model_config (ModelConfig): Model configuration, used for dtype.
        rank (int): Distributed rank.
        tp_rank (int): Tensor parallelism rank.
        world_size (int): Distributed world size.
    """

    def __init__(
        self,
        num_gpu_blocks: int = 128,
        rank: int = 0,
        tp_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.tp_rank = tp_rank

        # 使用 feature_dtype 替代 kv_cache_dtype，更符合场景
        self.feature_dtype = torch.float16
        self._num_gpu_blocks = num_gpu_blocks

        # 初始化 GPU 上的图像特征池
        self.gpu_feature_pool = self._allocate_gpu_cache()
        
        self.migration_backend_impl: Optional[MigrationBackendImpl] = None

        # 用于缓存操作的 CUDA 流
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # 用于流同步的事件
        self.events = torch.cuda.Event()

        logger.debug(f'Initialize feature cache engine with {self.num_gpu_blocks} gpu blocks.')

    @property
    def gpu_cache(self) -> torch.Tensor:
        """The GPU feature pool tensor."""
        return self.gpu_feature_pool

    @property
    def num_gpu_blocks(self) -> int:
        """Number of GPU blocks."""
        return self._num_gpu_blocks

    @staticmethod
    def get_feature_block_shape() -> Tuple[int, int]:
        """Get the shape of a single image feature block."""
        return FEATURE_BLOCK_SHAPE

    def _allocate_cache(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        """Allocate the memory pool on the specified device."""
        block_shape = self.get_feature_block_shape()

        # 直接分配一个大的、连续的 Tensor 作为特征池
        feature_pool = torch.empty(
            size=(num_blocks, *block_shape),
            dtype=self.feature_dtype,
            device=device,
        )
        return feature_pool

    def _allocate_gpu_cache(self) -> torch.Tensor:
        """Allocate the feature pool on the GPU."""
        return self._allocate_cache(self.num_gpu_blocks, 'cuda')

    @classmethod
    def get_cache_block_size(cls, model_config: ModelConfig) -> int:
        """Get the memory size in bytes of a single feature block.
        
        Args:
            model_config (ModelConfig): The model config, used for dtype.

        Return:
            int: Required memory size in bytes for one block.
        """
        shape = cls.get_feature_block_shape()
        dtype = model_config.dtype
        
        # 创建一个元张量以计算大小，避免实际内存分配
        meta_tensor = torch.empty(shape, dtype=dtype, device='meta')
        return meta_tensor.numel() * meta_tensor.element_size()

    """ Methods for Disaggregation Begin. """

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> List[DistServeKVTransferEndpointInfo]:
        if not self.migration_backend_impl:
            self.migration_backend_impl : MigrationBackendImpl = MIGRATION_BACKENDS.module_dict['DLSlime']()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)
        
        t = self.gpu_feature_pool
        if t.numel() > 0:
            register_mr_request = DistServeRegisterMRMessage(
                protocol=migration_init_request.protocol,
                remote_engine_id=migration_init_request.remote_engine_id,
                mr_key="feature_pool",  # 使用固定 key
                addr=t.data_ptr(),
                offset=t.storage_offset(),
                length=t.numel() * t.itemsize
            )
            self.migration_backend_impl.register_memory_region(register_mr_request)
            
        return [DistServeKVTransferEndpointInfo(
            protocol=migration_init_request.protocol,
            endpoint_info=json.dumps(
                self.migration_backend_impl.endpoint_info(
                    migration_init_request.remote_engine_id,
                    migration_init_request.protocol
                )
            )
        )]

    def p2p_connect(self, remote_engine_id: str, migration_conn_request: List[DistServeKVTransferEndpointInfo]):
        self.migration_backend_impl.p2p_connect(remote_engine_id, migration_conn_request[self.tp_rank])

    """ Methods for Disaggregation End. """