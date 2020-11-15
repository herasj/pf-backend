import { UserSchema } from 'src/schemas/user.schema';
import { UsersController } from './users.controller';
import { MongooseModule } from '@nestjs/mongoose';
import { UsersService } from './users.service';
import { Module } from '@nestjs/common';

@Module({
  imports: [MongooseModule.forFeature([{ name: 'users', schema: UserSchema }]),],
  controllers: [UsersController],
  providers: [UsersService],
})
export class UsersModule {}
