# ==========================================================
# Réorganisation du module market_structure
# ==========================================================

Write-Host "Création des dossiers..."

New-Item -ItemType Directory -Force src\market_structure\detectors | Out-Null
New-Item -ItemType Directory -Force src\market_structure\engines | Out-Null
New-Item -ItemType Directory -Force src\market_structure\pipelines | Out-Null
New-Item -ItemType Directory -Force src\market_structure\tests | Out-Null

New-Item -ItemType File -Force src\market_structure\detectors\__init__.py | Out-Null
New-Item -ItemType File -Force src\market_structure\engines\__init__.py | Out-Null
New-Item -ItemType File -Force src\market_structure\pipelines\__init__.py | Out-Null
New-Item -ItemType File -Force src\market_structure\tests\__init__.py | Out-Null

Write-Host "Déplacement des detectors..."

Move-Item src\market_structure\swing_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\trend_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\bos_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\choch_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\equal_high_low_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\liquidity_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\fair_value_gap_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\order_block_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\support_resistance_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\premium_discount_detector.py src\market_structure\detectors\ -Force
Move-Item src\market_structure\session_detector.py src\market_structure\detectors\ -Force

Write-Host "Déplacement des engines..."

Move-Item src\market_structure\bos_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\choch_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\liquidity_pool_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\fair_value_gap_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\order_block_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\order_block_lifecycle_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\market_structure_score_engine.py src\market_structure\engines\ -Force
Move-Item src\market_structure\trade_setup_engine.py src\market_structure\engines\ -Force

Write-Host "Déplacement du pipeline et du test..."

Move-Item src\market_structure\market_structure_pipeline.py src\market_structure\pipelines\ -Force
Move-Item src\market_structure\test_market_structure.py src\market_structure\tests\ -Force

Write-Host "Mise à jour des imports..."

Get-ChildItem src -Recurse -Filter *.py | ForEach-Object {
    $file = $_.FullName

    (Get-Content $file) `
    -replace "src\.market_structure\.swing_detector", "src.market_structure.detectors.swing_detector" `
    -replace "src\.market_structure\.trend_detector", "src.market_structure.detectors.trend_detector" `
    -replace "src\.market_structure\.bos_detector", "src.market_structure.detectors.bos_detector" `
    -replace "src\.market_structure\.choch_detector", "src.market_structure.detectors.choch_detector" `
    -replace "src\.market_structure\.equal_high_low_detector", "src.market_structure.detectors.equal_high_low_detector" `
    -replace "src\.market_structure\.liquidity_detector", "src.market_structure.detectors.liquidity_detector" `
    -replace "src\.market_structure\.fair_value_gap_detector", "src.market_structure.detectors.fair_value_gap_detector" `
    -replace "src\.market_structure\.order_block_detector", "src.market_structure.detectors.order_block_detector" `
    -replace "src\.market_structure\.support_resistance_detector", "src.market_structure.detectors.support_resistance_detector" `
    -replace "src\.market_structure\.premium_discount_detector", "src.market_structure.detectors.premium_discount_detector" `
    -replace "src\.market_structure\.session_detector", "src.market_structure.detectors.session_detector" `
    -replace "src\.market_structure\.bos_engine", "src.market_structure.engines.bos_engine" `
    -replace "src\.market_structure\.choch_engine", "src.market_structure.engines.choch_engine" `
    -replace "src\.market_structure\.liquidity_pool_engine", "src.market_structure.engines.liquidity_pool_engine" `
    -replace "src\.market_structure\.fair_value_gap_engine", "src.market_structure.engines.fair_value_gap_engine" `
    -replace "src\.market_structure\.order_block_engine", "src.market_structure.engines.order_block_engine" `
    -replace "src\.market_structure\.order_block_lifecycle_engine", "src.market_structure.engines.order_block_lifecycle_engine" `
    -replace "src\.market_structure\.market_structure_score_engine", "src.market_structure.engines.market_structure_score_engine" `
    -replace "src\.market_structure\.trade_setup_engine", "src.market_structure.engines.trade_setup_engine" `
    -replace "src\.market_structure\.market_structure_pipeline", "src.market_structure.pipelines.market_structure_pipeline" `
    | Set-Content $file
}

Write-Host "Réorganisation terminée."
Write-Host "Teste maintenant avec :"
Write-Host "python -m src.market_structure.tests.test_market_structure"