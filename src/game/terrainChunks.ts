/**
 * Infinite chunked voxel terrain for RATFIRE.
 *
 * The original map was a single 128x128 merged voxel mesh; this module turns
 * the exact same generator into an endless streaming world:
 *
 *  - DETERMINISTIC HEIGHT FIELD: `blockHeight(gx, gz)` reproduces the original
 *    4-octave ImprovedNoise sum per block (quality 2/8/32/128, `*0.15 | 0`)
 *    using a per-session seed. The grid origin keeps its +64 offset, so the
 *    spawn area is IDENTICAL to the old fixed map — the world simply no
 *    longer ends at the old borders.
 *  - CHUNKS: the plane is cut into CHUNK_BLOCKS x CHUNK_BLOCKS block chunks,
 *    each merged into its own mesh (one draw call, frustum-culled). Side
 *    faces are generated from the global height function, so chunk seams are
 *    invisible and the streaming frontier never shows holes.
 *  - STREAMING: `update(playerX, playerZ)` keeps every chunk within
 *    `viewRadius` (Chebyshev distance) of the player loaded, builds missing
 *    ones nearest-first with a per-frame budget (no hitching), and disposes
 *    chunks that fall more than one ring behind — the terrain behind you is
 *    freed as new terrain materialises ahead of you.
 *  - COLLISION: `surfaceYAt(worldX, worldZ)` answers from the deterministic
 *    height function, so walking, gravity, fall damage and the camera work
 *    at any distance from the origin, even over not-yet-meshed ground.
 */

import * as THREE from 'three';
import { ImprovedNoise } from 'three/examples/jsm/math/ImprovedNoise.js';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export interface TerrainChunksOptions {
  /** World units per voxel block edge (100). */
  block: number;
  /** Grid origin offset (64) — keeps spawn-area terrain identical to the
   *  original fixed 128x128 map (world x = (gx - gridOffset) * block). */
  gridOffset: number;
  /** Blocks per chunk side. */
  chunkBlocks?: number;
  /** Loaded-chunk radius around the player, in chunks (Chebyshev distance). */
  viewRadius?: number;
  /** ImprovedNoise z coordinate — the per-session terrain seed. */
  seedZ: number;
  /** Mobile LOW tier: smaller radius + build budget. */
  lowSpec?: boolean;
}

export interface TerrainChunksHandle {
  /** Stream around the player: build ahead, unload behind. Call per frame. */
  update(playerX: number, playerZ: number): void;
  /** Immediately (sync=true) or prioritised (sync=false) load the chunks
   *  within `radius` chunks of the point — used for spawn + respawn. */
  ensureAround(playerX: number, playerZ: number, radius: number, sync: boolean): void;
  /** Deterministic integer block height at grid coordinates. */
  blockHeight(gx: number, gz: number): number;
  /** Terrain surface height (top of the block) at a world-space position. */
  surfaceYAt(worldX: number, worldZ: number): number;
  /** Loaded chunk count + build-queue length (debug handle). */
  stats(): { chunks: number; pending: number };
  dispose(): void;
}

/** Height-cache cap: beyond this many memoised blocks the cache resets
 *  (a cache miss is only four ImprovedNoise calls, so this is invisible). */
const HEIGHT_CACHE_CAP = 250_000;

export function createTerrainChunks(
  scene: THREE.Scene,
  options: TerrainChunksOptions
): TerrainChunksHandle {
  const block = options.block;
  const gridOffset = options.gridOffset;
  const chunkBlocks = options.chunkBlocks ?? 16;
  const viewRadius = Math.max(2, options.viewRadius ?? 6);
  const buildBudget = options.lowSpec ? 1 : 2; // chunk meshes built per frame

  // ---------------- deterministic height field ----------------
  const perlin = new ImprovedNoise();
  const seedZ = options.seedZ;
  const heightCache = new Map<string, number>();

  function blockHeight(gx: number, gz: number): number {
    const key = gx + ',' + gz;
    const cached = heightCache.get(key);
    if (cached !== undefined) return cached;

    // exact reproduction of the original generateHeight() octave stack
    let h = 0;
    let quality = 2;
    for (let o = 0; o < 4; o++) {
      h += perlin.noise(gx / quality, gz / quality, seedZ) * quality;
      quality *= 4;
    }
    const height = (h * 0.15) | 0;

    if (heightCache.size > HEIGHT_CACHE_CAP) heightCache.clear();
    heightCache.set(key, height);
    return height;
  }

  /** Terrain surface height (top of the block) at a world-space position. */
  function surfaceYAt(worldX: number, worldZ: number): number {
    const gx = Math.round(worldX / block) + gridOffset;
    const gz = Math.round(worldZ / block) + gridOffset;
    return blockHeight(gx, gz) * block + block / 2;
  }

  // ---------------- shared voxel face prototypes ----------------
  // Same geometry + UV tweaks as the original demo (atlas-mapped textures).
  const matrix = new THREE.Matrix4();

  const pxGeometry = new THREE.PlaneGeometry(block, block);
  pxGeometry.attributes.uv.array[1] = 0.5;
  pxGeometry.attributes.uv.array[3] = 0.5;
  pxGeometry.rotateY(Math.PI / 2);
  pxGeometry.translate(block / 2, 0, 0);

  const nxGeometry = new THREE.PlaneGeometry(block, block);
  nxGeometry.attributes.uv.array[1] = 0.5;
  nxGeometry.attributes.uv.array[3] = 0.5;
  nxGeometry.rotateY(-Math.PI / 2);
  nxGeometry.translate(-block / 2, 0, 0);

  const pyGeometry = new THREE.PlaneGeometry(block, block);
  pyGeometry.attributes.uv.array[5] = 0.5;
  pyGeometry.attributes.uv.array[7] = 0.5;
  pyGeometry.rotateX(-Math.PI / 2);
  pyGeometry.translate(0, block / 2, 0);

  const pzGeometry = new THREE.PlaneGeometry(block, block);
  pzGeometry.attributes.uv.array[1] = 0.5;
  pzGeometry.attributes.uv.array[3] = 0.5;
  pzGeometry.translate(0, 0, block / 2);

  const nzGeometry = new THREE.PlaneGeometry(block, block);
  nzGeometry.attributes.uv.array[1] = 0.5;
  nzGeometry.attributes.uv.array[3] = 0.5;
  nzGeometry.rotateY(Math.PI);
  nzGeometry.translate(0, 0, -block / 2);

  const texture = new THREE.TextureLoader().load(
    '/textures/minecraft/atlas.png'
  );
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;

  const material = new THREE.MeshLambertMaterial({
    map: texture,
    side: THREE.DoubleSide,
  });

  // ---------------- chunk store ----------------
  interface Chunk {
    cx: number;
    cz: number;
    mesh: THREE.Mesh;
  }
  const chunks = new Map<string, Chunk>();

  function chunkKey(cx: number, cz: number): string {
    return cx + ',' + cz;
  }

  /** Builds one chunk mesh (merged voxel faces) and adds it to the scene. */
  function buildChunk(cx: number, cz: number): void {
    const key = chunkKey(cx, cz);
    if (chunks.has(key)) return;

    const baseX = cx * chunkBlocks;
    const baseZ = cz * chunkBlocks;
    const parts: THREE.BufferGeometry[] = [];

    for ( let lz = 0; lz < chunkBlocks; lz++ ) {
      for ( let lx = 0; lx < chunkBlocks; lx++ ) {
        const gx = baseX + lx;
        const gz = baseZ + lz;
        const h = blockHeight(gx, gz);

        matrix.makeTranslation(
          (gx - gridOffset) * block,
          h * block,
          (gz - gridOffset) * block
        );

        // neighbour heights come from the global function, so faces at
        // chunk borders match the adjacent chunk exactly (no seams/holes)
        const px = blockHeight(gx + 1, gz);
        const nx = blockHeight(gx - 1, gz);
        const pz = blockHeight(gx, gz + 1);
        const nz = blockHeight(gx, gz - 1);

        parts.push(pyGeometry.clone().applyMatrix4(matrix));

        if (px !== h && px !== h + 1) {
          parts.push(pxGeometry.clone().applyMatrix4(matrix));
        }
        if (nx !== h && nx !== h + 1) {
          parts.push(nxGeometry.clone().applyMatrix4(matrix));
        }
        if (pz !== h && pz !== h + 1) {
          parts.push(pzGeometry.clone().applyMatrix4(matrix));
        }
        if (nz !== h && nz !== h + 1) {
          parts.push(nzGeometry.clone().applyMatrix4(matrix));
        }
      }
    }

    const merged = BufferGeometryUtils.mergeGeometries(parts);
    for (const part of parts) part.dispose();
    if (!merged) return;
    merged.computeBoundingSphere();

    const mesh = new THREE.Mesh(merged, material);
    mesh.castShadow = true; // hills cast shadows into valleys
    mesh.receiveShadow = true; // player shadow lands here
    mesh.matrixAutoUpdate = false; // static geometry, world-space vertices
    scene.add(mesh);

    chunks.set(key, { cx, cz, mesh });
  }

  function unloadChunk(key: string, chunk: Chunk): void {
    scene.remove(chunk.mesh);
    chunk.mesh.geometry.dispose();
    chunks.delete(key);
  }

  /** Player chunk coordinates from a world position. */
  function playerChunk(playerX: number, playerZ: number): { cx: number; cz: number } {
    const gx = playerX / block + gridOffset;
    const gz = playerZ / block + gridOffset;
    return {
      cx: Math.floor(gx / chunkBlocks),
      cz: Math.floor(gz / chunkBlocks),
    };
  }

  /**
   * Streams the world around (pcx, pcz): builds up to `budget` missing
   * chunks (nearest first), unloads everything beyond viewRadius + 1.
   */
  let pendingCount = 0;

  function stream(pcx: number, pcz: number, budget: number): void {
    // --- collect missing chunks inside the view radius, nearest first ---
    const missing: Array<{ cx: number; cz: number; d: number }> = [];
    for (let dz = -viewRadius; dz <= viewRadius; dz++) {
      for (let dx = -viewRadius; dx <= viewRadius; dx++) {
        const cx = pcx + dx;
        const cz = pcz + dz;
        if (!chunks.has(chunkKey(cx, cz))) {
          missing.push({ cx, cz, d: Math.max(Math.abs(dx), Math.abs(dz)) });
        }
      }
    }
    pendingCount = missing.length;
    if (missing.length > 0) {
      missing.sort((a, b) => a.d - b.d);
      for (let i = 0; i < Math.min(budget, missing.length); i++) {
        buildChunk(missing[i].cx, missing[i].cz);
      }
    }

    // --- unload terrain that fell out of the keep-radius behind the player ---
    const keep = viewRadius + 1;
    for (const [key, chunk] of chunks) {
      const d = Math.max(
        Math.abs(chunk.cx - pcx),
        Math.abs(chunk.cz - pcz)
      );
      if (d > keep) unloadChunk(key, chunk);
    }
  }

  // ---------------- public API ----------------

  function update(playerX: number, playerZ: number): void {
    const { cx, cz } = playerChunk(playerX, playerZ);
    // Runs EVERY frame on purpose: the per-frame build budget is what lets
    // the frontier catch up after a chunk crossing adds a whole new row of
    // missing chunks (the scan itself is only ~a hundred map lookups).
    stream(cx, cz, buildBudget);
  }

  function ensureAround(
    playerX: number,
    playerZ: number,
    radius: number,
    sync: boolean
  ): void {
    const { cx, cz } = playerChunk(playerX, playerZ);
    if (!sync) return; // non-sync is covered by the per-frame update()
    // synchronous: every missing chunk in the radius is built right now,
    // guaranteeing ground under the player (spawn, respawn, teleports)
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dx = -radius; dx <= radius; dx++) {
        buildChunk(cx + dx, cz + dz);
      }
    }
  }

  function stats(): { chunks: number; pending: number } {
    return { chunks: chunks.size, pending: pendingCount };
  }

  function dispose(): void {
    for (const chunk of chunks.values()) {
      scene.remove(chunk.mesh);
      chunk.mesh.geometry.dispose();
    }
    chunks.clear();
    heightCache.clear();
    material.dispose();
    texture.dispose();
  }

  return {
    update,
    ensureAround,
    blockHeight,
    surfaceYAt,
    stats,
    dispose,
  };
}
