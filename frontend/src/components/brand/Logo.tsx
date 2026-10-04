import logoUrl from '../../assets/dnagen-logo.webp'
import { BRAND_NAME } from '../../lib/brand'

/** Helix + wordmark. The image is 332×120 (about 3× its display size) so it stays sharp on high-DPI screens. */
export function Logo({ className = 'h-10 w-auto' }: { className?: string }) {
  return <img src={logoUrl} alt={BRAND_NAME} width={332} height={120} decoding="async" className={className} />
}
