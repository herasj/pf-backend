import { IsBoolean, IsMongoId, IsNotEmpty } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class UpdatePoliticalTweetDTO {
  @IsNotEmpty()
  @IsMongoId()
  @ApiProperty({ type: String })
  _id: string;

  @IsNotEmpty()
  @IsBoolean()
  @ApiProperty({ type: Boolean })
  political: boolean;
}
